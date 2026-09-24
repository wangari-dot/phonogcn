# KSL-Daily-500 (full-coverage synthetic placeholder)

> ⚠️ **Clip data is fully synthetic.** See `SYNTHETIC_DATA_NOTICE.txt`.
> Every entry is flagged `"synthetic": true`. The gloss vocabulary is the
> 607-gloss **expert-validated** list
> (`cleaned_expert_validated.csv` via `build_vocab.py`); its assignment to
> these synthetic clips is procedural. Generated 900 clips so that **all
> 607 glosses** appear in training.

| Field | Value |
|---|---|
| Total clips | 900 |
| Train / Val / Test | 630 / 135 / 135 (signer-independent) |
| Signers (synthetic) | 12 |
| Gloss vocabulary | 607 |
| Glosses covered in train | 607 |
| Clip duration | 1.92–5.4 s |
| FPS | 25 |
| Skeleton | (T, 32, 3) float32 |
| Generator | dataset_toolkit/make_synthetic_dataset.py, seed 42 |
