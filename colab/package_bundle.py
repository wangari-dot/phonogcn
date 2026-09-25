"""
Packs the PhonoGCN code + the real-clip dataset into phonogcn_colab.zip for
training on a Colab GPU.

Crops are trimmed to the first --max_frames frames while zipping: train.py
only ever reads min(len(skeleton), data.n_frames_max) frames, so the rest is
dead weight in the upload. If you raise n_frames_max, rebuild the bundle with
a matching --max_frames. Skeletons are kept whole (they are small).

Usage:
    python package_bundle.py --data_root ../KSL_Daily_500_dataset/KSL_Daily_500_real334 \
        --code_dir "../PhonoGCN model" --notebook run_phonogcn_colab.ipynb \
        --out "F:/paper 2/phonogcn_colab.zip"
"""
import argparse, io, json, os, zipfile
import numpy as np

CODE_KEEP = ("config.py", "train.py", "evaluate.py", "requirements.txt", "LICENSE", "README.md")
CODE_DIRS = ("model", "pipeline")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--code_dir", required=True)
    ap.add_argument("--notebook", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_frames", type=int, default=150)
    a = ap.parse_args()

    root = "phonogcn_colab"
    dname = os.path.basename(os.path.normpath(a.data_root))
    clips = []
    for split in ("train", "val", "test"):
        clips += [e["clip_id"] for e in json.load(open(os.path.join(a.data_root, f"{split}_manifest.json"), encoding="utf-8"))]

    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.write(a.notebook, f"{root}/{os.path.basename(a.notebook)}")
        for f in CODE_KEEP:
            p = os.path.join(a.code_dir, f)
            if os.path.exists(p):
                z.write(p, f"{root}/code/{f}")
        for d in CODE_DIRS:
            for f in sorted(os.listdir(os.path.join(a.code_dir, d))):
                if f.endswith(".py"):
                    z.write(os.path.join(a.code_dir, d, f), f"{root}/code/{d}/{f}")

        droot = f"{root}/data/{dname}"
        for f in ("train_manifest.json", "val_manifest.json", "test_manifest.json",
                  "gloss_vocabulary.json", "extraction_log.csv", "coverage.csv"):
            p = os.path.join(a.data_root, f)
            if os.path.exists(p):
                z.write(p, f"{droot}/{f}")
        for i, cid in enumerate(clips, 1):
            z.write(os.path.join(a.data_root, "skeletons", cid + ".npy"), f"{droot}/skeletons/{cid}.npy")
            for part in ("right_hand", "left_hand", "face"):
                arr = np.load(os.path.join(a.data_root, "crops", cid, part + ".npy"))[:a.max_frames]
                buf = io.BytesIO()
                np.save(buf, arr)
                z.writestr(f"{droot}/crops/{cid}/{part}.npy", buf.getvalue())
            if i % 50 == 0 or i == len(clips):
                print(f"  packed {i}/{len(clips)} clips", flush=True)

    print(f"wrote {a.out} ({os.path.getsize(a.out) / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
