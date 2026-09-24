#!/usr/bin/env python3
"""
04_build_manifests.py — Turn expert annotations + extracted skeletons into
train/val/test manifests in the exact KSL_Daily_500 schema.

Input annotation CSV (one row per clip; export from ELAN or your tracker):
    clip_id,gloss,gloss_sequence,text,signer_id,fps
    a3f7c1b2d4e5,HOSPITAL,PRESIDENT GO HOSPITAL TODAY,Rais alienda...,youla_n,25

Splits are SIGNER-INDEPENDENT: whole signers are held out for val/test so
models cannot pass by memorising interpreters. With few TV interpreters
this is non-negotiable — random clip splits inflate accuracy.

Usage:
    python 04_build_manifests.py --annotations annotations.csv \
        --skeletons_dir skeletons/ --vocab gloss_vocabulary.json \
        --out_dir dataset/ --val_signers youla_n --test_signers wilson_m
"""
import argparse, csv, json, os
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--skeletons_dir", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--val_signers", nargs="+", default=[])
    ap.add_argument("--test_signers", nargs="+", default=[])
    a = ap.parse_args()

    vocab = json.load(open(a.vocab, encoding="utf-8"))
    gid = {v["gloss"]: v["id"] for v in vocab}
    os.makedirs(a.out_dir, exist_ok=True)
    manifests = {"train": [], "val": [], "test": []}
    skipped = []

    with open(a.annotations, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = row["clip_id"].strip()
            sk_path = os.path.join(a.skeletons_dir, cid + ".npy")
            if not os.path.exists(sk_path):
                skipped.append((cid, "no skeleton file")); continue
            seq = row["gloss_sequence"].split()
            unknown = [g for g in seq if g not in gid]
            if unknown:
                skipped.append((cid, f"glosses not in vocab: {unknown}")); continue
            sk = np.load(sk_path)
            fps = float(row.get("fps", 25) or 25)
            signer = row["signer_id"].strip()
            split = ("val" if signer in a.val_signers else
                     "test" if signer in a.test_signers else "train")
            manifests[split].append({
                "clip_id": cid, "gloss": row["gloss"].strip(),
                "gloss_sequence": " ".join(seq),
                "gloss_ids": [gid[g] for g in seq],
                "text": row["text"].strip(), "split": split,
                "signer_id": signer,
                "duration_sec": round(len(sk) / fps, 2),
                "n_frames": int(len(sk)), "fps": int(fps),
                "n_joints": int(sk.shape[1]),
                "synthetic": False, "pseudo_weight": 1.0,
                "source": "broadcast",
                "skeleton_path": f"skeletons/{cid}.npy",
                "crops_path": f"crops/{cid}/",
            })

    for split, entries in manifests.items():
        out = os.path.join(a.out_dir, f"{split}_manifest.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=1, ensure_ascii=False)
        print(f"{split}: {len(entries)} clips -> {out}")
    if skipped:
        with open(os.path.join(a.out_dir, "skipped.csv"), "w", newline="",
                  encoding="utf-8") as f:
            csv.writer(f).writerows([("clip_id", "reason")] + skipped)
        print(f"skipped {len(skipped)} rows — see skipped.csv")

if __name__ == "__main__":
    main()
