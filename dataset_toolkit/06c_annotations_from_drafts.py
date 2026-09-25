#!/usr/bin/env python3
"""
06c_annotations_from_drafts.py — Build annotations.csv for EVERY programme
clip in clips_annotated_v1.csv, not just the expert example clips.

Each clip's gloss_sequence is its gloss_draft with out-of-vocabulary tokens
dropped (order kept). Clips left with no in-vocabulary token, and rows with
qc_status=rejected (adverts), are skipped. Unlike 06_build_vocab_and_
annotations.py, a clip is NOT dropped just because its draft contains an
OOV token — the model then sees signs it has no label for, which is label
noise, but it gives every vocabulary gloss several clips across signers.

The draft is machine-made from the ASR transcript and never checked against
the video, so rows are pseudo-labels (pseudo_weight 0.4 by default).

Usage:
    python 06c_annotations_from_drafts.py \
        --vocab ../KSL_Daily_500_dataset/gloss_vocabulary.json \
        --clips_csv <metadata>/clips_annotated_v1.csv \
        --fps 10 --out ../annotations_all_10fps.csv
"""
import argparse, collections, csv, json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--clips_csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, required=True,
                    help="frame rate the skeletons were extracted at")
    ap.add_argument("--pseudo_weight", type=float, default=0.4)
    a = ap.parse_args()

    vocab = {v["gloss"] for v in json.load(open(a.vocab, encoding="utf-8"))}
    rows, n_rejected, n_empty, n_oov = [], 0, 0, 0
    with open(a.clips_csv, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["qc_status"].strip() == "rejected":
                n_rejected += 1
                continue
            draft = (row.get("gloss_draft") or "").split()
            seq = [t for t in draft if t in vocab]
            n_oov += len(draft) - len(seq)
            if not seq:
                n_empty += 1
                continue
            rows.append({
                "clip_id": row["clip_id"].strip(),
                "gloss": seq[0],
                "gloss_sequence": " ".join(seq),
                "text": (row.get("translation_en") or "").strip(),
                "signer_id": (row.get("signer_id") or "").strip(),
                "fps": a.fps,
                "source": "draft_in_vocab_tokens",
                "pseudo_weight": a.pseudo_weight,
            })

    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    cov = collections.Counter(g for r in rows for g in set(r["gloss_sequence"].split()))
    print(f"{len(rows)} clips -> {a.out}")
    print(f"skipped: {n_rejected} rejected (adverts), {n_empty} with no vocabulary token")
    print(f"dropped {n_oov} out-of-vocabulary draft tokens")
    print(f"glosses covered: {len(cov)}/{len(vocab)}, with >=5 clips: {sum(v >= 5 for v in cov.values())}")
    print("signers:", dict(collections.Counter(r["signer_id"] for r in rows)))


if __name__ == "__main__":
    main()
