# PhonoGCN Data Collection Pipeline — Technical Guide

This guide walks through the semi-automated pipeline used to construct
the KSL-Daily-500 benchmark from broadcast television footage.  The pipeline
is released in lieu of the raw dataset, which is withheld under the
Kenya Data Protection Act 2019.

---

## Overview

The pipeline has four sequential stages:

```
Broadcast video (KBC / Citizen TV)
        │
        ▼
[Stage 1]  broadcast_alignment.py   ← keyword spotting (TCN)
        │  Output: candidate clips + timestamps
        ▼
[Stage 2]  clustering.py            ← ST-GCN embedding + K-means
        │  Output: pseudo-labelled clips
        ▼
[Stage 3]  expert_validation.py     ← negative-selection review
        │  Output: validated clip manifest
        ▼
[Stage 4]  augmentation.py          ← geometric / temporal / skeletal aug
           Output: augmented dataset ready for training
```

Estimated throughput: ~2 hours of broadcast video → 500 validated sentences
(your results will vary depending on broadcast content and signer density).

---

## Prerequisites

Install all Python dependencies:

```bash
pip install -r requirements.txt
```

You will also need:
- **MediaPipe** (installed via requirements.txt) for pose extraction
- An **ASR backend** (Whisper recommended) for Stage 1 audio transcription
- Pre-trained checkpoints (see §6 below)

---

## Stage 1 — Broadcast Alignment

### What it does
Runs ASR on the audio track to obtain a timestamped transcript, then uses
a 2-layer TCN keyword spotter (pre-trained on WLASL-2000, fine-tuned on 80
KSL seed clips) to identify video segments containing target glosses.

### Running

```bash
python -m pipeline.broadcast_alignment \
    --video_dir   /path/to/broadcast_videos/ \
    --output_dir  output/candidates/ \
    --seed_model  checkpoints/seed_spotter.pt \
    --vocabulary  data/gloss_vocabulary.json \
    --threshold   0.5 \
    --clip_window 2.0 \
    --stride      0.5 \
    --device      cuda
```

### Key parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--threshold` | 0.5 | Detection threshold τ_det. See Appendix B.1 for sensitivity. |
| `--clip_window` | 2.0 s | Sliding window duration. |
| `--stride` | 0.5 s | Sliding window step. |

### Output
- `output/candidates/<video_stem>_candidates.json` — per-video candidate lists
- `output/candidates/all_candidates.json` — aggregated candidate list

Each candidate entry:
```json
{
  "gloss":      "HOSPITAL",
  "start_sec":  12.500,
  "end_sec":    14.500,
  "confidence": 0.823,
  "video_path": "/path/to/video.mp4"
}
```

### ASR integration
`run_asr()` in `broadcast_alignment.py` is a stub.  Replace it with your
preferred backend.  Example using OpenAI Whisper:

```python
import whisper

def run_asr(audio_path: str):
    model = whisper.load_model("medium")
    result = model.transcribe(audio_path, word_timestamps=True)
    words = []
    for seg in result["segments"]:
        for w in seg.get("words", []):
            words.append({"word": w["word"], "start": w["start"], "end": w["end"]})
    return words
```

---

## Stage 2 — Clustering & Pseudo-labelling

### What it does
Extracts ST-GCN embeddings from candidate clips, groups them per gloss
using K-means (K=3), and assigns high-confidence pseudo-labels to clips
within 1.5 standard deviations of their cluster centroid.

### Pre-requisite: skeleton extraction
Before running clustering, extract skeleton `.npy` files for each candidate:

```bash
python scripts/extract_skeletons.py \
    --candidates output/candidates/all_candidates.json \
    --output_dir output/skeletons/
```

*(Implement `scripts/extract_skeletons.py` using `PoseExtractor` from Stage 1.)*

### Running

```bash
python -m pipeline.clustering \
    --candidates   output/candidates/all_candidates.json \
    --skeleton_dir output/skeletons/ \
    --output       output/pseudo_labels.json \
    --encoder_ckpt checkpoints/stgcn_pretrained.pt \
    --n_clusters   3 \
    --sd_threshold 1.5 \
    --device       cuda
```

### Key parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--n_clusters` | 3 | K-means clusters per gloss group. |
| `--sd_threshold` | 1.5 | High-confidence radius in cluster SDs. See Appendix B.2. |

---

## Stage 3 — Expert Validation

### What it does
Routes pseudo-labelled candidates to KSL expert reviewers through a
**negative-selection** interface: reviewers reject incorrect clips rather
than confirming each one, reducing review burden for high-precision batches.

Inter-rater agreement is measured on a random 17.4% IRA subset using Cohen's κ.

### Running the data manager

```bash
python -m pipeline.expert_validation \
    --pseudo_labels  output/pseudo_labels.json \
    --output_dir     output/validated/ \
    --reviews        output/reviewer_decisions.json \
    --ira_fraction   0.174 \
    --min_reviewers  1 \
    --majority       0.5
```

### Review UI
A minimal Flask web interface for the negative-selection review workflow is
provided separately.  See `docs/review_ui_setup.md` for installation and
configuration instructions.

Alternatively, export the candidate list to any annotation tool that supports
video review (ELAN, Label Studio, CVAT) and import decisions as JSON:

```json
[
  {"clip_id": "a3f7c1b2d4e5", "reviewer_id": "expert_A", "accepted": true},
  {"clip_id": "b8e2a9c3f0d1", "reviewer_id": "expert_A", "accepted": false,
   "comment": "Wrong handshape — this is SCHOOL not HOSPITAL"}
]
```

### IRA interpretation
Cohen's κ values:
- κ < 0.40 — poor agreement; review annotation guidelines
- κ 0.40–0.60 — moderate
- κ 0.61–0.80 — substantial
- κ > 0.80 — near-perfect (**our result: κ = 0.82**)

---

## Stage 4 — Augmentation

### What it does
Applies geometric (rotation ±10°, scale 0.8–1.2×, horizontal flip),
temporal (speed perturbation, frame dropout), and skeletal (joint angle
perturbation ±15°) augmentation to reach the target 4.2× augmentation factor.
Rare-class clips (< 10 examples) receive additional skeletal perturbation.

### Running

```bash
python -m pipeline.augmentation \
    --skeleton_dir   output/skeletons/validated/ \
    --manifest       output/validated/validated_manifest.json \
    --output_dir     output/augmented/ \
    --augment_factor 4.2 \
    --rare_threshold 10 \
    --max_joint_deg  15.0 \
    --seed           42
```

---

## Data Layout (after pipeline)

```
data/ksl_daily_500/
├── skeletons/
│   ├── <clip_id>.npy        ← (T, J, 3) skeleton arrays
│   └── ...
├── crops/
│   ├── <clip_id>/
│   │   ├── right_hand/      ← (T,) directory of 112×112 PNG frames
│   │   ├── left_hand/
│   │   └── face/            ← 64×64 PNG frames
│   └── ...
├── train_manifest.json
├── val_manifest.json
└── test_manifest.json
```

Manifest format:
```json
[
  {
    "clip_id":   "a3f7c1b2d4e5",
    "gloss":     "HOSPITAL",
    "gloss_ids": [42],
    "text":      "Ana alienda hospitalini.",
    "split":     "train",
    "synthetic": false,
    "pseudo_weight": 1.0
  },
  ...
]
```

---

## Section 5 — Training

```bash
python train.py \
    --data_root data/ksl_daily_500 \
    --output_dir checkpoints/ \
    --device cuda
```

To override config hyperparameters without editing `config.py`:

```bash
python train.py \
    --data_root data/ksl_daily_500 \
    --output_dir checkpoints/ \
    --config_override train.lr=5e-5 train.batch_size=8
```

Training runs over 5 random seeds by default.  Results are reported as mean ± std.

---

## Section 6 — Evaluation

Single checkpoint:
```bash
python evaluate.py \
    --checkpoint checkpoints/seed_0/best_model.pt \
    --data_root  data/ksl_daily_500 \
    --split      test \
    --output     results/test_metrics.json
```

All seeds (reports mean ± std):
```bash
python evaluate.py \
    --all_seeds  checkpoints/ \
    --data_root  data/ksl_daily_500 \
    --split      test \
    --output     results/all_seeds_summary.json
```

---

## Section 7 — Pre-trained Checkpoints

| File | Description |
|------|-------------|
| `checkpoints/seed_spotter.pt` | KSL keyword spotter (Stage 1); WLASL-2000 pretrained, 80-clip KSL fine-tune |
| `checkpoints/stgcn_pretrained.pt` | ST-GCN body encoder; pretrained on WLASL-2000 + BSL-1K |

**Availability:** Checkpoints will be provided upon reasonable request to the
corresponding author.  They are not included in this repository due to file
size constraints.

---

## Section 8 — Extending the Vocabulary

To add new gloss classes:
1. Add the new gloss tokens to `data/gloss_vocabulary.json`
2. Collect ≥ 10 seed clips per new gloss
3. Fine-tune the keyword spotter on the expanded seed set (Stage 1)
4. Re-run the full pipeline
5. Re-train PhonoGCN with the updated vocabulary

---

## Section 9 — Citation

If you use this pipeline in your research, please cite:

```bibtex
@inproceedings{phonogcn2025,
  title     = {PhonoGCN: Phonologically Informed Graph Convolutional Networks
               for Continuous Kenyan Sign Language Recognition},
  author    = {Author One and Author Two and Author Three},
  booktitle = {Proceedings of ACL 2025},
  year      = {2025},
}
```

---

## Licence
MIT — see `../LICENSE`

