# PhoneGCN: A Phonologically Informed Graph Convolutional Network for Kenyan Sign Language Recognition and Translation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1](https://img.shields.io/badge/PyTorch-2.1-red.svg)](https://pytorch.org/)

## Overview

PhoneGCN is a phonologically informed graph convolutional network for low-resource Kenyan Sign Language (KSL) recognition and translation. The framework encodes handshape, movement, location, and non-manual signals through a multi-stream architecture, augmenting the standard skeleton graph with a learnable **Phonological Adjacency Matrix (PAM)** whose initial structure is derived from documented KSL co-articulatory dependencies.

**Key results on KSL-Daily-500:**
- **25.9 ± 0.8% WER** (22.3 pp improvement over I3D baseline)
- **5.8 pp improvement** over CorrNet (p < 0.01)
- **18.40 BLEU-4** for gloss-to-text translation

**Dataset Availability:** The KSL-Daily-607 dataset is provided in this repository as pre-extracted `.npy` feature files (MediaPipe pose, hand, and face keypoints). Raw video recordings are not released due to participant privacy obligations and ongoing community consultation. Researchers can use the included feature files directly for training and evaluation, or apply the provided pipeline code to collect equivalent datasets from broadcast media in their own contexts.

---

## Repository Structure

```
PhoneGCN/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── config.py                    # All hyperparameters and paths
├── train.py                     # Main training script
├── evaluate.py                  # Evaluation and ablation
│
├── pipeline/                    # Data collection pipeline
│   ├── __init__.py
│   ├── broadcast_alignment.py   # Keyword spotter (TCN on pose features)
│   ├── clustering.py            # K-means clustering + pseudo-labelling
│   ├── expert_validation.py     # Validation interface helpers
│   └── augmentation.py          # Geometric, temporal, skeletal augmentation
│
├── model/                       # Model architecture
│   ├── __init__.py
│   ├── phonogcn.py              # Top-level PhoneGCN model
│   ├── hand_stream.py           # EfficientNet-Lite hand stream
│   ├── body_stream.py           # ST-GCN body stream
│   ├── face_stream.py           # I3D face stream
│   ├── pam.py                   # Phonological Adjacency Matrix
│   ├── fusion.py                # Transformer fusion encoder
│   └── translation.py           # mBART gloss-to-text translation
│
└── docs/
    └── pipeline_guide.md        # Detailed pipeline usage guide
```

---

## Installation

```bash
git clone https://github.com/wangari-dot/PhonoGCN.git
cd PhoneGCN

# Create virtual environment
python -m venv venv
source venv/bin/activate      # Linux/macOS
# venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt

# Install MediaPipe (pose extraction)
pip install mediapipe==0.10.9
```

---

## Data Collection Pipeline

The semi-automated pipeline ingests broadcast media and produces a validated sign language dataset through three stages: weak supervision, unsupervised clustering, and expert validation.

```
Broadcast video
      │
      ▼
┌─────────────────────────────┐
│  1. Broadcast Alignment     │  ASR + keyword spotting (TCN)
│     broadcast_alignment.py  │  → 4,218 candidate clips
└─────────────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│  2. Clustering              │  K-means (k=3) + pseudo-labelling
│     clustering.py           │  → 2,874 high-confidence clips
└─────────────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│  3. Expert Validation       │  Negative-selection review
│     expert_validation.py    │  → 2,485 validated clips
└─────────────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│  4. Augmentation            │  Geometric + temporal + skeletal
│     augmentation.py         │  → ~4.2× effective training set
└─────────────────────────────┘
```

### Running the Pipeline

```bash
# Step 1: Extract candidate clips from broadcast recordings
python pipeline/broadcast_alignment.py \
    --video_dir /path/to/broadcasts \
    --output_dir /path/to/candidates \
    --threshold 0.5 \
    --seed_model checkpoints/seed_spotter.pt

# Step 2: Cluster and pseudo-label
python pipeline/clustering.py \
    --candidates_dir /path/to/candidates \
    --output_dir /path/to/pseudo_labelled \
    --encoder_checkpoint checkpoints/stgcn_pretrained.pt \
    --n_clusters 3 \
    --sd_threshold 1.5

# Step 3: Expert validation (generates review grids)
python pipeline/expert_validation.py \
    --pseudo_labelled_dir /path/to/pseudo_labelled \
    --output_review_dir /path/to/review

# Step 4: Augmentation (run after expert sign-off)
python pipeline/augmentation.py \
    --validated_dir /path/to/validated \
    --output_dir /path/to/augmented \
    --augment_factor 4.2
```

---

## Model Training

### Pretrain the encoder (contrastive self-supervised)

```bash
python train.py \
    --mode pretrain \
    --data_dir /path/to/augmented \
    --transfer_checkpoint checkpoints/wlasl2000_stgcn.pt \
    --epochs 60 \
    --lr 1e-4 \
    --batch_size 16 \
    --output_dir checkpoints/
```

### Fine-tune on KSL

```bash
python train.py \
    --mode finetune \
    --data_dir /path/to/augmented \
    --pretrain_checkpoint checkpoints/pretrained_encoder.pt \
    --epochs 60 \
    --lr 1e-4 \
    --alpha 0.3 \
    --beta 0.2 \
    --gamma 0.5 \
    --pseudo_discount 0.4 \
    --output_dir checkpoints/
```

### Train gloss-to-text translation

```bash
python train.py \
    --mode translation \
    --gloss_checkpoint checkpoints/best_recogniser.pt \
    --gloss_text_pairs /path/to/gloss_text.json \
    --mbart_model facebook/mbart-large-cc25 \
    --target_lang en_XX \
    --epochs 30 \
    --output_dir checkpoints/
```

---

## Evaluation

```bash
# Full evaluation (WER + BLEU-4 + ROUGE-L)
python evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --test_dir /path/to/test \
    --n_seeds 5

# Ablation study
python evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --test_dir /path/to/test \
    --ablation all

# Minimal pair analysis
python evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --minimal_pairs /path/to/minimal_pairs \
    --mode minimal_pairs
```

---

## Configuration

All hyperparameters are centralised in `config.py`. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr` | `1e-4` | Learning rate |
| `alpha` | `0.3` | Handshape loss weight |
| `beta` | `0.2` | Location loss weight |
| `gamma` | `0.5` | Contrastive loss weight |
| `delta` | `0.4` | Pseudo-label discount |
| `lambda_pam` | `0.5` | PAM balancing coefficient |
| `tau` | `0.07` | NT-Xent temperature (init) |
| `n_handshapes` | `38` | KSL handshape classes |
| `sd_threshold` | `1.5` | Pseudo-label SD boundary |
| `det_threshold` | `0.5` | Keyword spotter threshold |

---

## Citation

If you use this code or pipeline, please cite:

```bibtex
@article{phonogcn2026,
  title     = {Phono-GCN: A Phonologically Informed Graph Convolutional Network
               for Low-Resource Kenyan Sign Language Recognition and Translation},
  author    = {Author One and Author Two and Author Three},
  journal   = {[Under Review]},
  year      = {2026}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

This work builds on [MediaPipe](https://mediapipe.dev/), [ST-GCN](https://github.com/yysijie/st-gcn), [mBART](https://huggingface.co/facebook/mbart-large-cc25), and the KSL phonological analyses of Mweri (2018) and Okombo (1994).
