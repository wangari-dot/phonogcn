#!/usr/bin/env python3
"""
03_extract_skeletons.py — MediaPipe Holistic -> (T, 32, 3) skeleton .npy
matching the KSL_Daily_500 schema.

32-joint layout (default, override with --joint_map JSON):
    0-5   : pose landmarks 11-16 (shoulders, elbows, wrists)
    6-26  : right hand, all 21 landmarks
    27-31 : left hand summary (wrist 0, thumb tip 4, index tip 8,
            middle tip 12, pinky tip 20)

This layout privileges the dominant (right) hand, which suits TV
interpreters; adjust if your model expects a different graph. Missing
detections are zero-filled and per-clip detection rates are logged to
extraction_log.csv — clips under --min_hand_rate are moved to a
rejects list instead of the output folder.

Usage:
    python 03_extract_skeletons.py --crops_dir crops/ --out_dir skeletons/ \
        --min_hand_rate 0.6
"""
import argparse, csv, json, os
import numpy as np

RIGHT_SUMMARY = list(range(21))
LEFT_SUMMARY = [0, 4, 8, 12, 20]
POSE_IDX = [11, 12, 13, 14, 15, 16]

def extract(video, holistic, cv2):
    frames, hand_hits = [], 0
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        row = np.zeros((32, 3), dtype=np.float32)
        if res.pose_landmarks:
            for j, idx in enumerate(POSE_IDX):
                lm = res.pose_landmarks.landmark[idx]
                row[j] = (lm.x, lm.y, lm.z)
        if res.right_hand_landmarks:
            hand_hits += 1
            for j, idx in enumerate(RIGHT_SUMMARY):
                lm = res.right_hand_landmarks.landmark[idx]
                row[6 + j] = (lm.x, lm.y, lm.z)
        if res.left_hand_landmarks:
            for j, idx in enumerate(LEFT_SUMMARY):
                lm = res.left_hand_landmarks.landmark[idx]
                row[27 + j] = (lm.x, lm.y, lm.z)
        frames.append(row)
    cap.release()
    sk = np.stack(frames) if frames else np.zeros((0, 32, 3), np.float32)
    return sk, fps, hand_hits / max(len(frames), 1)

def main():
    import cv2, mediapipe as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--min_hand_rate", type=float, default=0.6)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.5,
                                              min_tracking_confidence=0.5)
    log_path = os.path.join(a.out_dir, "extraction_log.csv")
    with open(log_path, "a", newline="") as logf:
        log = csv.writer(logf)
        if logf.tell() == 0:
            log.writerow(["video", "n_frames", "fps", "hand_rate", "status"])
        for f in sorted(os.listdir(a.crops_dir)):
            if not f.lower().endswith((".mp4", ".mkv", ".webm")):
                continue
            path = os.path.join(a.crops_dir, f)
            sk, fps, rate = extract(path, holistic, cv2)
            stem = os.path.splitext(f)[0]
            if rate < a.min_hand_rate:
                log.writerow([f, len(sk), fps, f"{rate:.3f}", "REJECTED"])
                print(f"REJECT {f}: hand rate {rate:.1%}")
                continue
            np.save(os.path.join(a.out_dir, stem + ".npy"), sk)
            log.writerow([f, len(sk), fps, f"{rate:.3f}", "ok"])
            print(f"ok {f}: {len(sk)} frames, hands {rate:.1%}")

if __name__ == "__main__":
    main()
