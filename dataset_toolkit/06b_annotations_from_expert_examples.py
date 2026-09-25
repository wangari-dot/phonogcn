#!/usr/bin/env python3
"""
06b_annotations_from_expert_examples.py — Build annotations.csv for the
clips named in cleaned_expert_validated.csv (gloss -> example_clip_id).

Each validated clip becomes one annotation row. Its gloss_sequence is the
set of expert-validated glosses that point at that clip, ordered by where
they first occur in the clip's gloss_draft (clips_annotated_v1.csv). The
draft is machine-made from the ASR transcript, never checked against the
video, and other signs in the clip that are not in the vocabulary are
dropped — so these are pseudo-labels (source/pseudo_weight set to match
06_build_vocab_and_annotations.py), not video-verified annotation.

fps comes from the clip file itself (clips are a mix of 25 and 30 fps).

Usage:
    python 06b_annotations_from_expert_examples.py \
        --expert_csv ../cleaned_expert_validated.csv \
        --clips_csv  <metadata>/clips_annotated_v1.csv \
        --clips_dir  ../clips \
        --out        ../annotations_real334.csv
"""
import argparse, collections, csv, json, os, subprocess


def probe_fps(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=avg_frame_rate", "-of", "json", path],
        capture_output=True, text=True).stdout
    num, den = json.loads(out)["streams"][0]["avg_frame_rate"].split("/")
    return round(float(num) / float(den), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert_csv", required=True)
    ap.add_argument("--clips_csv", required=True)
    ap.add_argument("--clips_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pseudo_weight", type=float, default=0.4)
    a = ap.parse_args()

    by_clip = collections.defaultdict(list)
    with open(a.expert_csv, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            by_clip[row["example_clip_id"].strip()].append(row["Column1"].strip())

    meta = {}
    with open(a.clips_csv, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            meta[row["clip_id"].strip()] = row

    rows, unordered = [], 0
    for cid in sorted(by_clip):
        m = meta[cid]
        draft = (m.get("gloss_draft") or "").split()
        pos = {g: i for i, g in reversed(list(enumerate(draft)))}
        glosses = by_clip[cid]
        unordered += sum(g not in pos for g in glosses)
        seq = sorted(glosses, key=lambda g: pos.get(g, len(draft)))
        rows.append({
            "clip_id": cid,
            "gloss": seq[0],
            "gloss_sequence": " ".join(seq),
            "text": (m.get("translation_en") or "").strip(),
            "signer_id": (m.get("signer_id") or "").strip(),
            "fps": probe_fps(os.path.join(a.clips_dir, cid + ".mp4")),
            "source": "expert_vocab_example_draft_order",
            "pseudo_weight": a.pseudo_weight,
        })

    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"{len(rows)} clips, {sum(len(v) for v in by_clip.values())} glosses -> {a.out}")
    print(f"glosses not found in their clip's draft (appended at end): {unordered}")
    print("signers:", dict(collections.Counter(r["signer_id"] for r in rows)))


if __name__ == "__main__":
    main()
