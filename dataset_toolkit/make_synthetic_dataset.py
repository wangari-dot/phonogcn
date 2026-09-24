#!/usr/bin/env python3
"""
make_synthetic_dataset.py — Regenerate the KSL-Daily-500 synthetic placeholder
with FULL coverage of the gloss vocabulary (every gloss appears in training).
The vocabulary is the 607-gloss expert-validated list built by
build_vocab.py from cleaned_expert_validated.csv.

Output matches the exact schema of the original KSL_Daily_500 folder:
    skeletons/<clip_id>.npy            (T, 32, 3) float32
    crops/<clip_id>/{right_hand,left_hand,face}/features.npy   (T, 256) float32
    train/val/test_manifest.json       same keys as original
    gloss_vocabulary.json              copied from source
    dataset_card.md, SYNTHETIC_DATA_NOTICE.txt

EVERYTHING PRODUCED HERE IS SYNTHETIC. Every manifest entry carries
"synthetic": true and "source": "synthetic_placeholder". Do not remove
these flags: this data must never be presented as real KSL.

Design:
  - 12 synthetic signers, signer-independent 70/15/15 splits
    (signers 00-07 -> train, 08-09 -> val, 10-11 -> test).
  - Every vocabulary gloss appears as the primary gloss of >=1 training
    clip (so n_clips * 0.70 must be >= vocabulary size).
  - Skeletons: smooth procedural trajectories (sum of low-frequency
    sinusoids around signing-space rest pose). NOT real signing.
  - Hand/face features: random unit vectors.
  - Text: Swahili template substitution (lexicon fallback = gloss token).

Usage:
    python make_synthetic_dataset.py \
        --vocab ../KSL_Daily_500_dataset/gloss_vocabulary.json \
        --out   ../KSL_Daily_500_dataset/KSL_Daily_500_expert607 \
        --n_clips 900 --seed 42
"""
import argparse, hashlib, json, os, random, shutil
import numpy as np

FPS = 25
N_JOINTS = 32
FEAT_DIM = 256
DUR_RANGE = (1.92, 5.4)

# Small Swahili lexicon for template text (synthetic placeholder quality).
SW = {
    "PRESIDENT": "rais", "GOVERNMENT": "serikali", "POLICE": "polisi",
    "DOCTOR": "daktari", "HOSPITAL": "hospitali", "TODAY": "leo",
    "TOMORROW": "kesho", "YESTERDAY": "jana", "GO": "kwenda", "COME": "kuja",
    "SAY": "kusema", "ANNOUNCE": "kutangaza", "SCHOOL": "shule",
    "CHILD": "mtoto", "WATER": "maji", "MONEY": "pesa", "RAIN": "mvua",
    "COURT": "mahakama", "ARREST": "kukamatwa", "ACCIDENT": "ajali",
    "ROAD": "barabara", "COUNTRY": "nchi", "PEOPLE": "watu", "NEWS": "habari",
    "MEETING": "mkutano", "ELECTION": "uchaguzi", "TEACHER": "mwalimu",
    "FOOD": "chakula", "HELP": "kusaidia", "WORK": "kazi", "HOME": "nyumbani",
}

def sw(g): return SW.get(g, g.lower())

def clip_id(i, seed):
    return hashlib.md5(f"{seed}:{i}".encode()).hexdigest()[:12]

def make_skeleton(rng, n_frames):
    """Smooth pseudo-signing trajectories in normalised signing space."""
    t = np.linspace(0, 1, n_frames)[:, None, None]          # (T,1,1)
    rest = rng.uniform(-0.3, 0.5, size=(1, N_JOINTS, 3))     # rest pose
    amp = rng.uniform(0.02, 0.18, size=(3, N_JOINTS, 3))     # 3 harmonics
    freq = rng.uniform(0.5, 3.0, size=(3, 1, 1))
    phase = rng.uniform(0, 2 * np.pi, size=(3, N_JOINTS, 3))
    sk = rest + sum(amp[k] * np.sin(2 * np.pi * freq[k] * t + phase[k])
                    for k in range(3))
    sk += rng.normal(0, 0.004, sk.shape)                      # sensor jitter
    return np.clip(sk, -0.6, 0.7).astype(np.float32)

def make_features(rng, n_frames):
    f = rng.normal(size=(n_frames, FEAT_DIM)).astype(np.float32)
    return f / np.linalg.norm(f, axis=1, keepdims=True)

def make_text(seq):
    words = [sw(g) for g in seq]
    s = " ".join(words) + "."
    return s[0].upper() + s[1:]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_clips", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    vocab = json.load(open(args.vocab, encoding="utf-8"))
    glosses = [v["gloss"] for v in vocab]
    gid = {v["gloss"]: v["id"] for v in vocab}
    assert len(glosses) >= 1
    rng = np.random.default_rng(args.seed)
    pyrng = random.Random(args.seed)

    n = args.n_clips
    # Signer-independent split plan: 70/15/15.
    n_train, n_val = round(n * 0.70), round(n * 0.15)
    if n_train < len(glosses):
        raise SystemExit(f"train split ({n_train}) < vocabulary size "
                         f"({len(glosses)}); cannot cover every gloss — "
                         "increase --n_clips.")
    n_test = n - n_train - n_val
    signers = {"train": [f"signer_{i:02d}" for i in range(8)],
               "val":   ["signer_08", "signer_09"],
               "test":  ["signer_10", "signer_11"]}

    # Primary-gloss assignment: train covers ALL glosses, extras + val/test
    # sampled uniformly (all classes therefore appear in training).
    train_glosses = glosses + pyrng.choices(glosses, k=n_train - len(glosses))
    val_glosses = pyrng.choices(glosses, k=n_val)
    test_glosses = pyrng.choices(glosses, k=n_test)
    pyrng.shuffle(train_glosses)

    out = args.out
    os.makedirs(f"{out}/skeletons", exist_ok=True)
    os.makedirs(f"{out}/crops", exist_ok=True)

    manifests = {"train": [], "val": [], "test": []}
    i = 0
    for split, plist in [("train", train_glosses), ("val", val_glosses),
                         ("test", test_glosses)]:
        for g in plist:
            cid = clip_id(i, args.seed)
            dur = float(rng.uniform(*DUR_RANGE))
            T = int(round(dur * FPS))
            np.save(f"{out}/skeletons/{cid}.npy", make_skeleton(rng, T))
            for part in ("right_hand", "left_hand", "face"):
                d = f"{out}/crops/{cid}/{part}"
                os.makedirs(d, exist_ok=True)
                np.save(f"{d}/features.npy", make_features(rng, T))
            k = pyrng.randint(2, 5)
            seq = [g] + pyrng.sample([x for x in glosses if x != g], k)
            pyrng.shuffle(seq)
            manifests[split].append({
                "clip_id": cid, "gloss": g,
                "gloss_sequence": " ".join(seq),
                "gloss_ids": [gid[x] for x in seq],
                "text": make_text(seq), "split": split,
                "signer_id": pyrng.choice(signers[split]),
                "duration_sec": round(dur, 2), "n_frames": T, "fps": FPS,
                "n_joints": N_JOINTS, "synthetic": True, "pseudo_weight": 1.0,
                "source": "synthetic_placeholder",
                "skeleton_path": f"skeletons/{cid}.npy",
                "crops_path": f"crops/{cid}/",
            })
            i += 1

    for split, entries in manifests.items():
        with open(f"{out}/{split}_manifest.json", "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=1, ensure_ascii=False)
    shutil.copy(args.vocab, f"{out}/gloss_vocabulary.json")

    src_dir = os.path.dirname(os.path.abspath(args.vocab))
    for aux in ("SYNTHETIC_DATA_NOTICE.txt",):
        p = os.path.join(src_dir, aux)
        if os.path.exists(p):
            shutil.copy(p, f"{out}/{aux}")

    covered = {e["gloss"] for e in manifests["train"]}
    card = f"""# KSL-Daily-500 (full-coverage synthetic placeholder)

> ⚠️ **Clip data is fully synthetic.** See `SYNTHETIC_DATA_NOTICE.txt`.
> Every entry is flagged `"synthetic": true`. The gloss vocabulary is the
> {len(glosses)}-gloss **expert-validated** list
> (`cleaned_expert_validated.csv` via `build_vocab.py`); its assignment to
> these synthetic clips is procedural. Generated {n} clips so that **all
> {len(glosses)} glosses** appear in training.

| Field | Value |
|---|---|
| Total clips | {n} |
| Train / Val / Test | {n_train} / {n_val} / {n_test} (signer-independent) |
| Signers (synthetic) | 12 |
| Gloss vocabulary | {len(glosses)} |
| Glosses covered in train | {len(covered)} |
| Clip duration | {DUR_RANGE[0]}–{DUR_RANGE[1]} s |
| FPS | {FPS} |
| Skeleton | (T, {N_JOINTS}, 3) float32 |
| Generator | dataset_toolkit/make_synthetic_dataset.py, seed {args.seed} |
"""
    with open(f"{out}/dataset_card.md", "w", encoding="utf-8") as f:
        f.write(card)
    print(f"done: {n} clips, {len(covered)}/{len(glosses)} glosses in train -> {out}")

if __name__ == "__main__":
    main()
