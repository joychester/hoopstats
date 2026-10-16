"""第 1 步：整段视频跑检测 + 估计镜头运动，结果缓存到 JSON（改规则不需要重跑模型）。

用法: python detect_video.py VIDEO OUT.json [--step 2] [--model models/yolox_m.onnx]
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

from detector import Detector
from frames import iter_frames, video_info


def camera_motion(prev_gray, gray, mask_boxes):
    """估计相邻两帧之间的镜头运动（相似变换），只用背景特征点（排除人所在区域）。"""
    mask = np.full(prev_gray.shape, 255, np.uint8)
    for x1, y1, x2, y2 in mask_boxes:
        mask[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)] = 0
    pts = cv2.goodFeaturesToTrack(prev_gray, 400, 0.01, 8, mask=mask)
    if pts is None or len(pts) < 20:
        return np.eye(2, 3).tolist()
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None)
    good = st.flatten() == 1
    if good.sum() < 20:
        return np.eye(2, 3).tolist()
    M, _ = cv2.estimateAffinePartial2D(pts[good], nxt[good], method=cv2.RANSAC)
    return (M if M is not None else np.eye(2, 3)).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--step", type=int, default=2, help="detect every N frames (2 at 30fps = 15fps)")
    ap.add_argument("--model", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "yolox_m.onnx"))
    a = ap.parse_args()

    det = Detector(a.model)
    fps, W, H = video_info(a.video)
    frames, prev_gray, prev_boxes, t0 = [], None, [], time.time()
    t_model = t_motion = 0.0
    scale = 0.5  # 运动估计用半分辨率
    for i, img in iter_frames(a.video, a.step):
        ta = time.time()
        gray = cv2.cvtColor(cv2.resize(img, None, fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)
        M = np.eye(2, 3).tolist()
        if prev_gray is not None:
            M = camera_motion(prev_gray, gray, [[v * scale for v in b] for b in prev_boxes])
            M[0][2] /= scale
            M[1][2] /= scale
        tb = time.time()
        dets = det.detect(img, conf_person=0.25, conf_ball=0.1)
        tc = time.time()
        t_motion += tb - ta
        t_model += tc - tb
        frames.append({"frame": i, "t": round(i / fps, 3), "motion": M,
                       "dets": [[c, round(s, 3)] + [round(v, 1) for v in b] for c, s, *b in dets]})
        prev_gray, prev_boxes = gray, [d[2:] for d in dets if d[0] == 0]
        if len(frames) % 50 == 0:
            print(f"  detecting: {i / fps:.0f}s of video processed, {time.time() - t0:.0f}s elapsed", flush=True)
    if not frames:
        raise SystemExit(f"could not read any frames from {a.video} (unsupported or damaged file)")
    json.dump({"video": a.video, "fps": fps, "width": W, "height": H, "step": a.step,
               "frames": frames}, open(a.out, "w"))
    total = time.time() - t0
    print(f"done: {len(frames)} frames in {total:.0f}s "
          f"(model {t_model:.0f}s, camera motion {t_motion:.0f}s, decoding {total - t_model - t_motion:.0f}s)")


if __name__ == "__main__":
    main()
