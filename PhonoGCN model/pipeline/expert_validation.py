"""
expert_validation.py
====================
Stage 3 of the PhoneGCN data collection pipeline.

Presents pseudo-labelled candidate clips to KSL expert reviewers through
a negative-selection interface: experts reject incorrect clips rather than
confirming each one, making review faster for high-precision batches.

Inter-rater agreement is measured on a random 17.4% subset (500 of 2,874
pseudo-labelled clips) using Cohen's κ.  Full corpus agreement is estimated
from this sample (κ = 0.82; see §3.1.3).

NOTE
----
This module provides the review data-management layer.  The actual web UI
is a lightweight Flask app (see docs/pipeline_guide.md §3.3).  If you
prefer to integrate into an existing annotation tool (ELAN, Label Studio,
etc.) you can use ``ExpertValidationSession`` as a data adapter.
"""

import os
import json
import random
import logging
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

class ReviewRecord:
    """Immutable record of a single expert's decision on one clip."""

    def __init__(self, clip_id: str, reviewer_id: str,
                 accepted: bool, timestamp: Optional[str] = None,
                 comment: str = ""):
        self.clip_id    = clip_id
        self.reviewer_id = reviewer_id
        self.accepted   = accepted
        self.timestamp  = timestamp or datetime.now(timezone.utc).isoformat()
        self.comment    = comment

    def to_dict(self) -> Dict:
        return {
            "clip_id":     self.clip_id,
            "reviewer_id": self.reviewer_id,
            "accepted":    self.accepted,
            "timestamp":   self.timestamp,
            "comment":     self.comment,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "ReviewRecord":
        return cls(
            clip_id=d["clip_id"],
            reviewer_id=d["reviewer_id"],
            accepted=d["accepted"],
            timestamp=d.get("timestamp"),
            comment=d.get("comment", ""),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Agreement metrics
# ─────────────────────────────────────────────────────────────────────────────

def cohen_kappa(rater_a: List[bool], rater_b: List[bool]) -> float:
    """
    Compute Cohen's κ for binary ratings.

    Args:
        rater_a: Binary decisions from reviewer A (True = accept).
        rater_b: Binary decisions from reviewer B (True = accept).

    Returns:
        κ ∈ [-1, 1]; κ = 0 indicates chance agreement.
    """
    if len(rater_a) != len(rater_b) or len(rater_a) == 0:
        raise ValueError("Rater lists must be non-empty and equal length.")

    n = len(rater_a)
    agree = sum(a == b for a, b in zip(rater_a, rater_b))
    p_o = agree / n

    p_a_yes = sum(rater_a) / n
    p_b_yes = sum(rater_b) / n
    p_e = p_a_yes * p_b_yes + (1 - p_a_yes) * (1 - p_b_yes)

    if p_e == 1.0:
        return 1.0  # perfect by chance — trivial case
    return (p_o - p_e) / (1 - p_e)


def compute_pairwise_kappa(reviews: Dict[str, List[ReviewRecord]]
                           ) -> Dict[Tuple[str, str], float]:
    """
    Compute κ for every reviewer pair on their shared clip subset.

    Args:
        reviews: {clip_id: [ReviewRecord, ...]}

    Returns:
        {(reviewer_a, reviewer_b): kappa_value}
    """
    # Build {clip_id: {reviewer_id: accepted}}
    clip_decisions: Dict[str, Dict[str, bool]] = {}
    for clip_id, records in reviews.items():
        clip_decisions[clip_id] = {r.reviewer_id: r.accepted for r in records}

    # Collect all reviewer pairs
    all_reviewers = sorted({r.reviewer_id
                             for records in reviews.values()
                             for r in records})
    kappas = {}
    for i, ra in enumerate(all_reviewers):
        for rb in all_reviewers[i + 1:]:
            shared = [cid for cid, d in clip_decisions.items()
                      if ra in d and rb in d]
            if len(shared) < 2:
                continue
            a_ratings = [clip_decisions[cid][ra] for cid in shared]
            b_ratings = [clip_decisions[cid][rb] for cid in shared]
            kappas[(ra, rb)] = cohen_kappa(a_ratings, b_ratings)

    return kappas


# ─────────────────────────────────────────────────────────────────────────────
# Validation session manager
# ─────────────────────────────────────────────────────────────────────────────

class ExpertValidationSession:
    """
    Manages the negative-selection expert review workflow.

    Workflow:
        1. Load pseudo-labelled candidates.
        2. Sample the inter-rater agreement (IRA) subset (~17% of clips).
        3. Assign clips to reviewers (round-robin or manual assignment).
        4. Record accept / reject decisions.
        5. Resolve IRA subset via majority vote.
        6. Export validated dataset manifest.

    Usage::

        session = ExpertValidationSession(
            pseudo_labels_path="output/pseudo_labels.json",
            output_dir="output/validated/",
            ira_sample_fraction=0.174,
        )
        session.assign_reviewers(["expert_A", "expert_B"])
        # ... reviewers submit decisions via record_decision() ...
        session.export_validated_manifest()
    """

    def __init__(self,
                 pseudo_labels_path: str,
                 output_dir: str,
                 ira_sample_fraction: float = 0.174,
                 random_seed: int = 42):
        self.output_dir = output_dir
        self.ira_fraction = ira_sample_fraction
        self.random_seed = random_seed
        os.makedirs(output_dir, exist_ok=True)

        with open(pseudo_labels_path) as f:
            self.candidates: List[Dict] = json.load(f)

        # Assign deterministic clip IDs
        for cand in self.candidates:
            if "clip_id" not in cand:
                cand["clip_id"] = _make_clip_id(cand)

        # {clip_id: [ReviewRecord]}
        self.reviews: Dict[str, List[ReviewRecord]] = {
            c["clip_id"]: [] for c in self.candidates
        }

        self.ira_clip_ids: List[str] = []
        self.reviewer_assignments: Dict[str, List[str]] = {}  # reviewer → [clip_id]

        review_cache = os.path.join(output_dir, "reviews.json")
        if os.path.exists(review_cache):
            self._load_reviews(review_cache)
            logger.info(f"Resumed session from {review_cache}")

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def assign_reviewers(self, reviewer_ids: List[str]):
        """
        Assign clips to reviewers.

        The IRA subset is assigned to ALL reviewers.  Remaining clips are
        distributed round-robin.

        Args:
            reviewer_ids: List of reviewer identifiers (e.g. ["expert_A", "expert_B"]).
        """
        rng = random.Random(self.random_seed)
        all_ids = [c["clip_id"] for c in self.candidates if c.get("keep", True)]

        n_ira = max(2, int(len(all_ids) * self.ira_fraction))
        self.ira_clip_ids = rng.sample(all_ids, n_ira)
        ira_set = set(self.ira_clip_ids)
        remaining = [cid for cid in all_ids if cid not in ira_set]

        # All reviewers see the IRA subset
        self.reviewer_assignments = {r: list(self.ira_clip_ids) for r in reviewer_ids}

        # Distribute remaining clips round-robin
        for i, cid in enumerate(remaining):
            reviewer = reviewer_ids[i % len(reviewer_ids)]
            self.reviewer_assignments[reviewer].append(cid)

        self._save_assignments()
        total = sum(len(v) for v in self.reviewer_assignments.values())
        logger.info(
            f"Assigned {len(self.ira_clip_ids)} IRA clips + "
            f"{len(remaining)} non-IRA clips → {total} total review tasks "
            f"across {len(reviewer_ids)} reviewers"
        )

    # ------------------------------------------------------------------
    # Recording decisions
    # ------------------------------------------------------------------

    def record_decision(self, clip_id: str, reviewer_id: str,
                        accepted: bool, comment: str = ""):
        """
        Record one reviewer's accept/reject decision for a clip.

        Args:
            clip_id:     Clip identifier.
            reviewer_id: Reviewer identifier.
            accepted:    True = clip is a valid example of the labelled gloss.
            comment:     Optional free-text note.
        """
        if clip_id not in self.reviews:
            raise KeyError(f"Unknown clip_id: {clip_id}")

        record = ReviewRecord(clip_id, reviewer_id, accepted, comment=comment)
        self.reviews[clip_id].append(record)
        self._save_reviews()

    def record_decisions_batch(self, decisions: List[Dict]):
        """
        Batch-record decisions from a list of dicts.

        Each dict must contain: clip_id, reviewer_id, accepted.
        Optional: comment.
        """
        for d in decisions:
            self.record_decision(
                clip_id=d["clip_id"],
                reviewer_id=d["reviewer_id"],
                accepted=d["accepted"],
                comment=d.get("comment", ""),
            )

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def compute_ira_kappa(self) -> Dict:
        """
        Compute Cohen's κ on the IRA subset for all reviewer pairs.

        Returns:
            {
                "n_ira_clips": int,
                "pairwise_kappa": {"{A}_{B}": float},
                "mean_kappa": float,
            }
        """
        ira_reviews = {cid: self.reviews[cid] for cid in self.ira_clip_ids}
        kappas = compute_pairwise_kappa(ira_reviews)
        result = {
            "n_ira_clips": len(self.ira_clip_ids),
            "pairwise_kappa": {f"{a}_{b}": round(k, 4)
                               for (a, b), k in kappas.items()},
            "mean_kappa": round(sum(kappas.values()) / max(len(kappas), 1), 4),
        }
        logger.info(f"IRA κ = {result['mean_kappa']} (n = {result['n_ira_clips']} clips)")
        return result

    def resolve_accepted_clips(self,
                               min_reviewers: int = 1,
                               majority_threshold: float = 0.5
                               ) -> List[str]:
        """
        Determine which clips pass expert validation.

        A clip is accepted if:
            - At least `min_reviewers` decisions have been recorded, AND
            - The fraction of "accept" decisions ≥ majority_threshold.

        Args:
            min_reviewers:       Minimum number of decisions required.
            majority_threshold:  Fraction threshold (default 0.5 = simple majority).

        Returns:
            List of accepted clip_ids.
        """
        accepted = []
        for cid, records in self.reviews.items():
            if len(records) < min_reviewers:
                continue
            accept_rate = sum(r.accepted for r in records) / len(records)
            if accept_rate >= majority_threshold:
                accepted.append(cid)

        logger.info(
            f"Accepted {len(accepted)} / {len(self.reviews)} clips "
            f"(min_reviewers={min_reviewers}, threshold={majority_threshold})"
        )
        return accepted

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_validated_manifest(self,
                                  min_reviewers: int = 1,
                                  majority_threshold: float = 0.5
                                  ) -> str:
        """
        Write the validated dataset manifest to output_dir/validated_manifest.json.

        Returns:
            Path to the written manifest file.
        """
        accepted_ids = set(self.resolve_accepted_clips(min_reviewers, majority_threshold))

        # Build clip_id lookup
        candidates_by_id = {c["clip_id"]: c for c in self.candidates}

        validated = []
        for cid in accepted_ids:
            entry = dict(candidates_by_id[cid])
            entry["review_records"] = [r.to_dict() for r in self.reviews[cid]]
            validated.append(entry)

        validated.sort(key=lambda x: (x["gloss"], x["start_sec"]))

        out_path = os.path.join(self.output_dir, "validated_manifest.json")
        with open(out_path, "w") as f:
            json.dump(validated, f, indent=2)

        ira_stats = self.compute_ira_kappa()
        stats = {
            "total_candidates": len(self.candidates),
            "reviewed": sum(1 for recs in self.reviews.values() if recs),
            "accepted": len(validated),
            "rejection_rate": round(1 - len(validated) / max(len(self.candidates), 1), 4),
            "ira": ira_stats,
        }
        stats_path = os.path.join(self.output_dir, "validation_stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Validated manifest → {out_path}")
        logger.info(f"Validation stats   → {stats_path}")
        return out_path

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _save_reviews(self):
        path = os.path.join(self.output_dir, "reviews.json")
        serialisable = {
            cid: [r.to_dict() for r in records]
            for cid, records in self.reviews.items()
        }
        with open(path, "w") as f:
            json.dump(serialisable, f, indent=2)

    def _load_reviews(self, path: str):
        with open(path) as f:
            raw = json.load(f)
        for cid, records in raw.items():
            if cid in self.reviews:
                self.reviews[cid] = [ReviewRecord.from_dict(r) for r in records]

    def _save_assignments(self):
        path = os.path.join(self.output_dir, "reviewer_assignments.json")
        with open(path, "w") as f:
            json.dump(self.reviewer_assignments, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _make_clip_id(candidate: Dict) -> str:
    """Generate a deterministic clip ID from video path + timestamps."""
    key = f"{candidate['video_path']}_{candidate['start_sec']}_{candidate['end_sec']}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="PhoneGCN Expert Validation — report IRA statistics"
    )
    parser.add_argument("--pseudo_labels",  required=True, help="pseudo_labels.json from Stage 2")
    parser.add_argument("--output_dir",     required=True, help="Directory to write validated manifest")
    parser.add_argument("--reviews",        required=True, help="JSON file of reviewer decisions")
    parser.add_argument("--ira_fraction",   type=float, default=0.174)
    parser.add_argument("--min_reviewers",  type=int,   default=1)
    parser.add_argument("--majority",       type=float, default=0.5)
    args = parser.parse_args()

    session = ExpertValidationSession(
        pseudo_labels_path=args.pseudo_labels,
        output_dir=args.output_dir,
        ira_sample_fraction=args.ira_fraction,
    )

    with open(args.reviews) as f:
        decisions = json.load(f)
    session.record_decisions_batch(decisions)

    session.export_validated_manifest(
        min_reviewers=args.min_reviewers,
        majority_threshold=args.majority,
    )
