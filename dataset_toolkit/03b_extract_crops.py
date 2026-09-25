#!/usr/bin/env python3
"""
03b_extract_crops.py — MediaPipe Holistic -> per-clip skeleton (T,32,3) +
right-hand/left-hand/face pixel crops (T,3,H,W), in one pass per video.

Combines what 03_extract_skeletons.py does (skeleton) with the raw
hand/face crops PhoneGCN's HandStream/FaceStream need, derived from the
SAME per-frame Holistic detections so each clip's video is decoded and
run through mediapipe only once. 03_extract_skeletons.py still works
standalone if only the skeleton is wanted.

Output layout (rooted at --out_dir, matching the skeleton_path/crops_path
convention 04_build_manifests.py writes into manifests):
    <out_dir>/skeletons/<clip_id>.npy          (T,32,3)     float32
    <out_dir>/crops/<clip_id>/right_hand.npy   (T,3,112,112) uint8
    <out_dir>/crops/<clip_id>/left_hand.npy    (T,3,112,112) uint8
    <out_dir>/crops/<clip_id>/face.npy         (T,3,64,64)   uint8

Detection rate is measured on the RIGHT hand (dominant hand for these
interpreters, same convention as 03_extract_skeletons.py) and logged to
extraction_log.csv; clips under --min_hand_rate are rejected (no output
written for that clip).

Usage:
    python 03b_extract_crops.py --clips_dir dataset/clips --out_dir dataset/keypoints \
        --min_hand_rate 0.6 [--target_fps 10]
"""
import argparse, csv, os
import numpy as np

RIGHT_SUMMARY = list(range(21))
LEFT_SUMMARY = [0, 4, 8, 12, 20]
POSE_IDX = [11, 12, 13, 14, 15, 16]

HAND_SIZE = 112
FACE_SIZE = 64
HAND_PAD = 0.4    # fraction of box size added on each side
FACE_PAD = 0.25


def _bbox_from_landmarks(landmarks, w, h, pad_frac):
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    x0, x1 = min(xs) * w, max(xs) * w
    y0, y1 = min(ys) * h, max(ys) * h
    bw, bh = x1 - x0, y1 - y0
    padx, pady = bw * pad_frac, bh * pad_frac
    x0, x1 = x0 - padx, x1 + padx
    y0, y1 = y0 - pady, y1 + pady
    # Square the box around its centre so resize doesn't distort aspect ratio.
    side = max(x1 - x0, y1 - y0, 1.0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0, x1 = cx - side / 2, cx + side / 2
    y0, y1 = cy - side / 2, cy + side / 2
    x0, y0 = max(0, int(round(x0))), max(0, int(round(y0)))
    x1, y1 = min(w, int(round(x1))), min(h, int(round(y1)))
    return x0, y0, x1, y1


def _crop_resize(frame, landmarks, pad_frac, out_size, cv2):
    h, w = frame.shape[:2]
    if landmarks is None:
        return np.zeros((3, out_size, out_size), dtype=np.uint8)
    x0, y0, x1, y1 = _bbox_from_landmarks(landmarks.landmark, w, h, pad_frac)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((3, out_size, out_size), dtype=np.uint8)
    patch = frame[y0:y1, x0:x1]
    patch = cv2.resize(patch, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    return patch.transpose(2, 0, 1)   # (3, out_size, out_size)


def extract(video_path, holistic, cv2, target_fps=None):
    sk_frames = []
    rh_frames, lh_frames, face_frames = [], [], []
    hand_hits = 0
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    # With target_fps, keep only the frames that start a new 1/target_fps
    # slot, so 25 and 30 fps sources both come out at the same rate.
    step = fps / target_fps if target_fps and target_fps < fps else 1.0
    i, next_keep = -1, 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        if i < next_keep - 1e-6:
            continue
        next_keep += step
        res = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        sk = np.zeros((32, 3), dtype=np.float32)
        if res.pose_landmarks:
            for j, idx in enumerate(POSE_IDX):
                lm = res.pose_landmarks.landmark[idx]
                sk[j] = (lm.x, lm.y, lm.z)
        if res.right_hand_landmarks:
            hand_hits += 1
            for j, idx in enumerate(RIGHT_SUMMARY):
                lm = res.right_hand_landmarks.landmark[idx]
                sk[6 + j] = (lm.x, lm.y, lm.z)
        if res.left_hand_landmarks:
            for j, idx in enumerate(LEFT_SUMMARY):
                lm = res.left_hand_landmarks.landmark[idx]
                sk[27 + j] = (lm.x, lm.y, lm.z)
        sk_frames.append(sk)

        rh_frames.append(_crop_resize(frame, res.right_hand_landmarks, HAND_PAD, HAND_SIZE, cv2))
        lh_frames.append(_crop_resize(frame, res.left_hand_landmarks, HAND_PAD, HAND_SIZE, cv2))
        face_frames.append(_crop_resize(frame, res.face_landmarks, FACE_PAD, FACE_SIZE, cv2))

    cap.release()
    n = len(sk_frames)
    skeleton = np.stack(sk_frames) if n else np.zeros((0, 32, 3), np.float32)
    right_hand = np.stack(rh_frames) if n else np.zeros((0, 3, HAND_SIZE, HAND_SIZE), np.uint8)
    left_hand = np.stack(lh_frames) if n else np.zeros((0, 3, HAND_SIZE, HAND_SIZE), np.uint8)
    face = np.stack(face_frames) if n else np.zeros((0, 3, FACE_SIZE, FACE_SIZE), np.uint8)
    rate = hand_hits / max(n, 1)
    return skeleton, right_hand, left_hand, face, fps / step, rate


def main():
    import cv2, mediapipe as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips_dir", required=True, nargs="+",
                    help="one or more folders of clip videos")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--min_hand_rate", type=float, default=0.6)
    ap.add_argument("--target_fps", type=float, default=None,
                    help="subsample every clip to this frame rate (e.g. 10)")
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N clips (smoke test)")
    ap.add_argument("--ids_file", default=None,
                    help="text file, one clip_id per line — process only "
                         "these clips (e.g. an incremental annotation batch)")
    a = ap.parse_args()

    skeletons_dir = os.path.join(a.out_dir, "skeletons")
    crops_root = os.path.join(a.out_dir, "crops")
    os.makedirs(skeletons_dir, exist_ok=True)
    os.makedirs(crops_root, exist_ok=True)

    want_ids = None
    if a.ids_file:
        with open(a.ids_file, encoding="utf-8") as f:
            want_ids = {line.strip() for line in f if line.strip()}

    holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.5,
                                              min_tracking_confidence=0.5)
    log_path = os.path.join(a.out_dir, "extraction_log.csv")
    write_header = not os.path.exists(log_path) or os.path.getsize(log_path) == 0
    with open(log_path, "a", newline="") as logf:
        log = csv.writer(logf)
        if write_header:
            log.writerow(["clip", "n_frames", "fps", "hand_rate", "status"])

        paths = {f: os.path.join(d, f) for d in a.clips_dir for f in os.listdir(d)
                 if f.lower().endswith((".mp4", ".mkv", ".webm"))}
        files = sorted(paths)
        if want_ids is not None:
            files = [f for f in files if os.path.splitext(f)[0] in want_ids]
        if a.limit:
            files = files[:a.limit]

        for f in files:
            path = paths[f]
            stem = os.path.splitext(f)[0]

            # Resume support: skip clips already fully extracted (all 4
            # files present) so re-running after an interruption doesn't
            # redo already-completed work.
            sk_path = os.path.join(skeletons_dir, stem + ".npy")
            clip_crop_dir = os.path.join(crops_root, stem)
            if (os.path.exists(sk_path)
                    and os.path.exists(os.path.join(clip_crop_dir, "right_hand.npy"))
                    and os.path.exists(os.path.join(clip_crop_dir, "left_hand.npy"))
                    and os.path.exists(os.path.join(clip_crop_dir, "face.npy"))):
                continue

            sk, rh, lh, face, fps, rate = extract(path, holistic, cv2, a.target_fps)

            if rate < a.min_hand_rate:
                log.writerow([f, len(sk), fps, f"{rate:.3f}", "REJECTED"])
                print(f"REJECT {f}: hand rate {rate:.1%}")
                continue

            np.save(os.path.join(skeletons_dir, stem + ".npy"), sk)
            os.makedirs(clip_crop_dir, exist_ok=True)
            np.save(os.path.join(clip_crop_dir, "right_hand.npy"), rh)
            np.save(os.path.join(clip_crop_dir, "left_hand.npy"), lh)
            np.save(os.path.join(clip_crop_dir, "face.npy"), face)

            log.writerow([f, len(sk), fps, f"{rate:.3f}", "ok"])
            print(f"ok {f}: {len(sk)} frames, hands {rate:.1%}")


if __name__ == "__main__":
    main()
