# KSL-Daily-500 dataset folders

**Current:** [`KSL_Daily_500_expert607/`](KSL_Daily_500_expert607/) — 900
clips, built on the **607-gloss expert-validated vocabulary**
(`../cleaned_expert_validated.csv` → `gloss_vocabulary.json` in this folder,
via `../dataset_toolkit/build_vocab.py`). All 607 glosses are covered in
training; splits are signer-independent (70/15/15).

> ⚠️ Clip-level data (skeletons, hand/face features, text) is still a
> **synthetic placeholder** for schema/pipeline testing — see
> `SYNTHETIC_DATA_NOTICE.txt`. Only the gloss vocabulary itself is
> expert-validated. Real clips come from the broadcast pipeline in
> `../dataset_toolkit/`.

The earlier placeholder folders built on the provisional 310-gloss
vocabulary (`KSL_Daily_500/`, `KSL_Daily_500_full310/`) have been removed;
`KSL_Daily_500_expert607/` supersedes them.

`gloss_vocabulary.json` at this level is the canonical vocabulary; the copy
inside each dataset folder records the vocabulary that folder was built with.
