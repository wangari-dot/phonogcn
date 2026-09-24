#!/usr/bin/env python3
"""
build_vocab.py — Build the canonical gloss vocabulary from the
expert-validated gloss list (cleaned_expert_validated.csv).

The CSV is the source of truth: one gloss label per row (single column,
header row ignored). This script:
  1. de-duplicates and sorts glosses alphabetically,
  2. assigns stable integer ids 0..N-1 in that order,
  3. writes gloss_vocabulary.json  ([{"id": i, "gloss": g}, ...]),
  4. writes elan_controlled_vocab.csv for the ELAN annotation tier.

Re-run whenever the expert list changes, then rebuild manifests
(04_build_manifests.py) and retrain with the new n_glosses.

Usage:
    python build_vocab.py \
        --csv  ../cleaned_expert_validated.csv \
        --json ../KSL_Daily_500_dataset/gloss_vocabulary.json \
        --elan elan_controlled_vocab.csv
"""
import argparse, csv, json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True,
                    help="expert-validated gloss list (one gloss per row)")
    ap.add_argument("--json", required=True,
                    help="output gloss_vocabulary.json path")
    ap.add_argument("--elan", help="optional output ELAN controlled-vocab CSV")
    args = ap.parse_args()

    with open(args.csv, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    glosses = [r[0].strip() for r in rows[1:] if r and r[0].strip()]
    seen, uniq = set(), []
    for g in glosses:
        if g not in seen:
            seen.add(g)
            uniq.append(g)
    uniq.sort()

    vocab = [{"id": i, "gloss": g} for i, g in enumerate(uniq)]
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)
    print(f"wrote {args.json}: {len(vocab)} glosses")

    if args.elan:
        with open(args.elan, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["value", "description"])
            for v in vocab:
                w.writerow([v["gloss"], f"id={v['id']}"])
        print(f"wrote {args.elan}: {len(vocab)} entries")


if __name__ == "__main__":
    main()
