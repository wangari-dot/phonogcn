#!/usr/bin/env python3
"""
06_build_vocab_and_annotations.py — Bootstrap gloss_vocabulary.json and
annotations.csv from the real KSL-Daily-500 data, without waiting on
per-clip video annotation.

gloss_vocabulary.json comes straight from the two-expert-validated
607-word list (metadata/cleaned_expert_validated.csv).

annotations.csv comes from clips_annotated_v1.csv's gloss_draft column
(machine-drafted from ASR transcripts, never checked against video) —
a row is kept only if EVERY draft token is in the validated vocabulary.
Kept rows are pseudo-labels, not expert annotation: source and
pseudo_weight are set accordingly so 04_build_manifests.py and the
model's training loss can discount them relative to real video-verified
clips (see config.py's delta=0.4).

Usage:
    python 06_build_vocab_and_annotations.py \
        --cleaned_vocab ../../../metadata/cleaned_expert_validated.csv \
        --clips_csv ../../../metadata/clips_annotated_v1.csv \
        --vocab_out ../gloss_vocabulary.json \
        --annotations_out ../annotations.csv
"""
import argparse, csv, json, collections


def load_vocab(path):
    words, seen = [], set()
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if not row:
                continue
            w = row[0].strip()
            if not w or w.lower() == "column1":
                continue
            if w not in seen:
                seen.add(w)
                words.append(w)
    return words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cleaned_vocab", required=True)
    ap.add_argument("--clips_csv", required=True)
    ap.add_argument("--vocab_out", required=True)
    ap.add_argument("--annotations_out", required=True)
    ap.add_argument("--pseudo_weight", type=float, default=0.4)
    ap.add_argument("--fps", type=float, default=25.0)
    a = ap.parse_args()

    words = load_vocab(a.cleaned_vocab)
    vocab = [{"gloss": w, "id": i} for i, w in enumerate(words)]
    with open(a.vocab_out, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=1, ensure_ascii=False)
    vocab_set = set(words)
    print(f"vocab: {len(words)} glosses -> {a.vocab_out}")

    kept, no_draft, oov_blocked = 0, 0, 0
    oov_counter = collections.Counter()
    rows_out = []
    with open(a.clips_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            draft = (row.get("gloss_draft") or "").strip()
            if not draft:
                no_draft += 1
                continue
            tokens = draft.split()
            unknown = [t for t in tokens if t not in vocab_set]
            if unknown:
                oov_blocked += 1
                oov_counter.update(unknown)
                continue
            kept += 1
            rows_out.append({
                "clip_id": row["clip_id"].strip(),
                "gloss": tokens[0],
                "gloss_sequence": " ".join(tokens),
                "text": (row.get("translation_en") or "").strip(),
                "signer_id": (row.get("signer_id") or "").strip(),
                "fps": a.fps,
                "source": "pseudo_labeled_vocab_validated",
                "pseudo_weight": a.pseudo_weight,
            })

    fieldnames = ["clip_id", "gloss", "gloss_sequence", "text", "signer_id",
                  "fps", "source", "pseudo_weight"]
    with open(a.annotations_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)

    total = kept + no_draft + oov_blocked
    print(f"clips_annotated_v1.csv rows: {total}")
    print(f"  kept (fully vocab-covered): {kept}")
    print(f"  no gloss_draft:             {no_draft}")
    print(f"  blocked by OOV token(s):    {oov_blocked}")
    print(f"-> {a.annotations_out}")
    print("\ntop 30 out-of-vocab tokens blocking the most rows "
          "(next expert-review batch):")
    for tok, cnt in oov_counter.most_common(30):
        print(f"  {tok:<20} {cnt}")


if __name__ == "__main__":
    main()
