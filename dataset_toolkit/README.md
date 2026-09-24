# KSL Broadcast Dataset Toolkit

Tools for building the real KSL-Daily-500 dataset from Citizen TV / KBC
bulletins, plus a generator for the synthetic placeholder. Companion to the
project plan (`KSL_Broadcast_Dataset_Project_Plan.docx`).

## What this toolkit does NOT do

- **Download broadcast footage.** You must obtain bulletins yourself and you
  are responsible for the legal basis (Kenya Copyright Act fair dealing for
  research covers analysis, not redistribution; YouTube ToS prohibits
  downloading — see plan §2.1). Send permission letters to RMS/KBC early.
- **Annotate.** Gloss labels require KSL-fluent annotators (plan §4.2).
  The repo's `pipeline/` (broadcast_alignment, clustering, expert_validation)
  provides the semi-automated candidate generation; humans still validate.

## Gloss vocabulary (expert-validated)

The canonical vocabulary is the **607-gloss expert-validated list** in
`../cleaned_expert_validated.csv`. `build_vocab.py` converts it into
`../KSL_Daily_500_dataset/gloss_vocabulary.json` (alphabetical, stable ids
0–606) and `elan_controlled_vocab.csv`. Re-run it if the expert list changes,
then rebuild manifests and update `n_glosses` in the model config.

## Pipeline order

| Step | Script | Output |
|---|---|---|
| 0a | `build_vocab.py` | Canonical `gloss_vocabulary.json` + ELAN controlled vocab from the expert-validated CSV |
| 0b | `make_synthetic_dataset.py` | Synthetic placeholder, full 607-gloss coverage (schema testing only) |
| 1 | *(you)* obtain bulletins | raw videos + a log of station/program/date |
| 2 | `02_crop_inset.py` | interpreter crops (444×444) + inset quality gate |
| 3 | `03_extract_skeletons.py` | `(T, 32, 3)` .npy per clip + detection log |
| 4 | *(annotators)* ELAN with `elan_controlled_vocab.csv` | annotations CSV |
| 5 | `04_build_manifests.py` | train/val/test manifests (signer-independent) |
| 6 | `05_coverage_report.py` | per-gloss progress vs 25-clip floor |

## The week-one pilot (do this before anything else)

1. Obtain 10 bulletins per station at max resolution.
2. `02_crop_inset.py --dump_frame` → measure inset box → `stations.json`.
3. Batch-crop, then `--check` each crop: hand-detection rate **< 60% means
   that station's inset is unusable** — stop and rethink before investing
   months (plan §2.2).
4. Have a KSL-fluent annotator tally which of the 607 glosses actually
   appear in 3–4 bulletins; feed the counts to `05_coverage_report.py`
   to size the studio top-up (plan §2.3).

## Integrity rules

- Synthetic clips carry `"synthetic": true` and
  `"source": "synthetic_placeholder"`. **Never strip these flags.** Results
  reported on synthetic data must be labeled as such.
- Real clips get `"source": "broadcast"` from `04_build_manifests.py`.
- Splits are signer-independent (whole interpreters held out). Random clip
  splits leak signer identity and produce numbers reviewers should not trust.

## 32-joint skeleton layout

`03_extract_skeletons.py` writes: pose 11–16 (shoulders/elbows/wrists) →
joints 0–5; right hand 21 landmarks → 6–26; left hand summary
(wrist, thumb/index/middle/pinky tips) → 27–31. If PhonoGCN's adjacency
assumes a different layout, edit `POSE_IDX` / `LEFT_SUMMARY` and keep
`model/pam.py` in sync.

## Dependencies

```
pip install numpy opencv-python mediapipe==0.10.9
# ffmpeg on PATH for cropping
```
