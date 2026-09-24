#!/usr/bin/env python3
"""
02_crop_inset.py — Crop the sign-language interpreter inset from broadcast
bulletins and check whether the inset is actually usable.

Requires: ffmpeg on PATH; opencv-python + mediapipe for the quality check.

The inset position is fixed per station/program, so you calibrate once per
station by inspecting a frame, then batch-crop. Station geometry lives in
stations.json:

    {
      "citizen_prime": {"x": 1520, "y": 780, "w": 380, "h": 290, "out": 444},
      "kbc_channel1":  {"x": 40,   "y": 800, "w": 360, "h": 270, "out": 444}
    }

Usage:
    # 1. Dump a frame to find the inset box (open frame.png, note pixels)
    python 02_crop_inset.py --video bulletin.mp4 --dump_frame frame.png

    # 2. Batch crop everything in a folder
    python 02_crop_inset.py --video_dir raw/ --station citizen_prime \
        --stations stations.json --out_dir crops/

    # 3. Quality gate: % of frames with detected hands in the crop
    python 02_crop_inset.py --check crops/bulletin_crop.mp4
"""
import argparse, json, os, subprocess, sys

def dump_frame(video, out_png, at="00:05:00"):
    subprocess.run(["ffmpeg", "-y", "-ss", at, "-i", video, "-frames:v", "1",
                    out_png], check=True)
    print(f"frame written to {out_png} — measure the inset box in pixels")

def crop(video, geom, out_path):
    vf = (f"crop={geom['w']}:{geom['h']}:{geom['x']}:{geom['y']},"
          f"scale={geom['out']}:{geom['out']}")
    subprocess.run(["ffmpeg", "-y", "-i", video, "-vf", vf, "-an",
                    "-c:v", "libx264", "-crf", "18", out_path], check=True)

def check(video, sample_every=5, max_frames=2000):
    """Hand-detection rate inside the crop. Below ~60% = inset unusable."""
    import cv2, mediapipe as mp
    holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.5)
    cap, n, hits, i = cv2.VideoCapture(video), 0, 0, 0
    while n < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        if i % sample_every:
            continue
        n += 1
        res = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.right_hand_landmarks or res.left_hand_landmarks:
            hits += 1
    cap.release()
    rate = hits / max(n, 1)
    print(f"{video}: hand detection {rate:.1%} over {n} sampled frames")
    if rate < 0.6:
        print("WARNING: below 60% — inset likely too small/occluded. "
              "Re-check crop geometry or drop this source.")
    return rate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video"); ap.add_argument("--video_dir")
    ap.add_argument("--dump_frame"); ap.add_argument("--check")
    ap.add_argument("--station"); ap.add_argument("--stations",
                                                  default="stations.json")
    ap.add_argument("--out_dir", default="crops")
    a = ap.parse_args()

    if a.dump_frame:
        return dump_frame(a.video, a.dump_frame)
    if a.check:
        return check(a.check)
    geom = json.load(open(a.stations))[a.station]
    vids = ([a.video] if a.video else
            [os.path.join(a.video_dir, f) for f in sorted(os.listdir(a.video_dir))
             if f.lower().endswith((".mp4", ".mkv", ".webm"))])
    os.makedirs(a.out_dir, exist_ok=True)
    for v in vids:
        out = os.path.join(a.out_dir,
                           os.path.splitext(os.path.basename(v))[0] + "_crop.mp4")
        print(f"cropping {v} -> {out}")
        crop(v, geom, out)

if __name__ == "__main__":
    sys.exit(main())
