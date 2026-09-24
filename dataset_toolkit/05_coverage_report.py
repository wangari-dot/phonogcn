#!/usr/bin/env python3
"""
05_coverage_report.py — Per-gloss clip counts against a target floor.

Run this weekly during annotation. It tells you which glosses are done,
which are on track, and which will never reach the floor from news footage
alone (candidates for studio top-up recording).

Usage:
    python 05_coverage_report.py --manifests dataset/ \
        --vocab gloss_vocabulary.json --floor 25 --csv coverage.csv
"""
import argparse, collections, csv, json, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", required=True,
                    help="dir containing *_manifest.json")
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--floor", type=int, default=25)
    ap.add_argument("--csv", help="optional output CSV path")
    a = ap.parse_args()

    vocab = [v["gloss"] for v in json.load(open(a.vocab, encoding="utf-8"))]
    counts = collections.Counter()
    for split in ("train", "val", "test"):
        p = os.path.join(a.manifests, f"{split}_manifest.json")
        if os.path.exists(p):
            for e in json.load(open(p, encoding="utf-8")):
                counts[e["gloss"]] += 1

    done = [g for g in vocab if counts[g] >= a.floor]
    partial = [g for g in vocab if 0 < counts[g] < a.floor]
    missing = [g for g in vocab if counts[g] == 0]

    print(f"floor: {a.floor} clips/gloss   vocabulary: {len(vocab)}")
    print(f"  done    (>= floor): {len(done)}")
    print(f"  partial (1..{a.floor - 1}):   {len(partial)}")
    print(f"  missing (0):        {len(missing)}")
    total_needed = sum(max(a.floor - counts[g], 0) for g in vocab)
    print(f"  clips still needed: {total_needed}")
    if missing:
        print("\nworst 20 (studio top-up candidates):")
        for g in sorted(vocab, key=lambda x: counts[x])[:20]:
            print(f"  {g:<20} {counts[g]}")

    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["gloss", "clips", "needed", "status"])
            for g in vocab:
                w.writerow([g, counts[g], max(a.floor - counts[g], 0),
                            "done" if counts[g] >= a.floor else
                            "partial" if counts[g] else "missing"])
        print(f"\nwrote {a.csv}")

if __name__ == "__main__":
    main()
